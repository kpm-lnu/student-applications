//
// Created by anmode on 27.08.2024.
//

#ifndef FEM_ENGINE_SHARED_REF_H
#define FEM_ENGINE_SHARED_REF_H

#include <stdexcept>

/**
     * \brief Base class for reference-object objects
     * Reference acquisition and release can be done manually by explicitly
     * calling add_ref() or remove_ref() on the reference object objects, or can be
     * done automatically by using SmartPointer<T>.
     * \see SmartPointer
     */

class SharedRefObject {
public:
    /**
     * \brief Increments the reference count
     */
    void add_ref() const {
        ++no_refs;
    }

    /**
     * \brief Decrements the reference count
     */
    void remove_ref() const {
        --no_refs;
        if (no_refs < 0) {
            throw std::runtime_error("Reference count is negative");
        }
        if(no_refs == 0) {
            delete this;
        }
    }

    /**
     * \brief Check if the object is shared
     * \return \c true if the object is shared, \c false otherwise
     */
    bool is_shared() const {
        return no_refs > 1;
    }

    /**
     * \brief Gets the number of references that point to this object.
     * \return the number of references.
     */
    int num_refs() const {
        return no_refs;
    }

    /**
     * \brief Increments the reference count
     */
    static void add_ref(const SharedRefObject* object) {
        if(object != nullptr) {
            object->add_ref();
        }
    }

    /**
     * \brief Decrements the reference count
     */
    static void remove_ref(const SharedRefObject* object) {
        if(object != nullptr) {
            object->remove_ref();
        }
    }

    protected:
    /**
     * \brief Creates a reference object object
     */
    SharedRefObject() :
    no_refs(0) {
    }

    /**
     * \brief Destroys a reference object object
     */
    virtual ~SharedRefObject();

private:
    /** Forbid copy constructor */
    SharedRefObject(const SharedRefObject&);
    /** Forbid assignment operator */
    SharedRefObject& operator= (const SharedRefObject&);

    mutable int no_refs;
};


#endif //FEM_ENGINE_SHARED_REF_H
